# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract for resolving many version selections out of ONE parsed corpus.

The fold used to run through the filesystem: load the staged file, merge one impl's
overlay, dump it back, repeat per version layer. It now folds in memory, and the
generator resolves ~20 version selections from a single `load_corpus()` parse. Two
things have to hold for that to be a pure speedup:

1. A resolved doc must not alias objects owned by the shared corpus. Overlay blocks
   are grafted into the base BY REFERENCE, so without a copy, selection A and
   selection B would hand back the same mutable objects and a reader that touched
   one would corrupt the other.
2. The copy must preserve object sharing WITHIN a source doc. Fixture files use YAML
   anchors (`&id002` / `*id002`) for repeated expected blocks; deep-copying each
   grafted value separately silently expanded every alias, so the staged file grew
   and stopped matching what the on-disk fold produced. Copying the doc in one call
   keeps the sharing — and the anchors — intact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

UTILS = Path(__file__).resolve().parents[1]
SRC = UTILS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import resolve_fixtures  # noqa: E402
import resolve_stream_fixtures  # noqa: E402
from fixture_corpus import load_corpus  # noqa: E402

FAMILY = "famx"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _batch_tree(root: Path) -> None:
    """Batch corpus whose overlay shares ONE expected block across two cases via a
    YAML anchor — the shape whose aliases the per-value copy expanded."""
    _write(root / "inputs" / FAMILY / "TOOLCALLING.batch.yaml", (
        "family: famx\n"
        "mode: batch\n"
        "cases:\n"
        "  TOOLCALLING.batch.1:\n"
        "    model_text: a\n"
        "  TOOLCALLING.batch.2:\n"
        "    model_text: b\n"
    ))
    for ver in ("0.23.0", "0.24.0"):
        _write(root / f"vllm_python-{ver}" / FAMILY / "TOOLCALLING.batch.yaml", (
            "family: famx\n"
            "mode: batch\n"
            "cases:\n"
            "  TOOLCALLING.batch.1:\n"
            "    expected:\n"
            "      vllm_python: &shared\n"
            "        calls: []\n"
            f"        normal_text: 'v{ver}'\n"
            "  TOOLCALLING.batch.2:\n"
            "    expected:\n"
            "      vllm_python: *shared\n"
        ))


def _stream_tree(root: Path) -> None:
    _write(root / "inputs" / FAMILY / "TOOLCALLING.streamv2.1.yaml", (
        "family: famx\n"
        "mode: streamv2\n"
        "cases:\n"
        "  TOOLCALLING.streamv2.1:\n"
        "    chunks:\n"
        "    - delta_text: 'x'\n"
        "    - delta_text: 'y'\n"
    ))
    for ver in ("0.1.11", "0.1.22"):
        _write(root / f"dynamo_v2-{ver}" / FAMILY / "TOOLCALLING.streamv2.1.yaml", (
            "family: famx\n"
            "mode: streamv2\n"
            "cases:\n"
            "  TOOLCALLING.streamv2.1:\n"
            "    chunks:\n"
            "    - expected: &shared\n"
            "      - index: 0\n"
            f"        name: 'v{ver}'\n"
            "    - expected: *shared\n"
        ))


RESOLVERS = {
    "batch": (resolve_fixtures, _batch_tree, "TOOLCALLING.batch.yaml",
              ["vllm_python-0.23.0"], ["vllm_python-0.24.0"]),
    "stream": (resolve_stream_fixtures, _stream_tree, "TOOLCALLING.streamv2.1.yaml",
               ["dynamo_v2-0.1.11"], ["dynamo_v2-0.1.22"]),
}


@pytest.mark.parametrize("kind", sorted(RESOLVERS))
def test_shared_corpus_matches_a_fresh_parse(tmp_path, kind):
    """resolve_docs(corpus=<shared>) == resolve_docs(corpus=None), for every selection."""
    mod, make_tree, name, sel_a, sel_b = RESOLVERS[kind]
    root = tmp_path / kind
    make_tree(root)
    corpus = load_corpus(root)

    for select in (sel_a, sel_b):
        shared, _ = mod.resolve_docs(root, select, corpus=corpus)
        fresh, _ = mod.resolve_docs(root, select)
        assert shared == fresh, f"{kind} {select}: shared-corpus resolve diverged"


@pytest.mark.parametrize("kind", sorted(RESOLVERS))
def test_selections_do_not_alias_each_other(tmp_path, kind):
    """Mutating one resolved selection must not reach another, or the corpus."""
    mod, make_tree, name, sel_a, sel_b = RESOLVERS[kind]
    root = tmp_path / kind
    make_tree(root)
    corpus = load_corpus(root)

    docs_a, _ = mod.resolve_docs(root, sel_a, corpus=corpus)
    docs_b, _ = mod.resolve_docs(root, sel_b, corpus=corpus)
    before_b = yaml.safe_dump(docs_b[(FAMILY, name)], sort_keys=False)
    before_corpus = yaml.safe_dump(
        {"/".join(k): v for k, v in sorted(corpus.items())}, sort_keys=False
    )

    # Reach into A and scribble on every mutable leaf a reader could touch.
    doc_a = docs_a[(FAMILY, name)]
    for case in (doc_a.get("cases") or {}).values():
        for blk in (case.get("expected") or {}).values():
            if isinstance(blk, dict):
                blk["normal_text"] = "SCRIBBLED"
        for ch in case.get("chunks") or []:
            for deltas in (ch.get("expected") or {}).values():
                if isinstance(deltas, list):
                    deltas.append({"index": 99, "name": "SCRIBBLED"})

    assert yaml.safe_dump(docs_b[(FAMILY, name)], sort_keys=False) == before_b, (
        f"{kind}: mutating one selection leaked into another"
    )
    assert yaml.safe_dump(
        {"/".join(k): v for k, v in sorted(corpus.items())}, sort_keys=False
    ) == before_corpus, f"{kind}: mutating a resolved selection leaked into the corpus"


@pytest.mark.parametrize("kind", sorted(RESOLVERS))
def test_source_anchors_survive_the_fold(tmp_path, kind):
    """Two cases sharing one object in the source must still share it after resolving,
    so the staged file re-emits the alias instead of expanding it."""
    mod, make_tree, name, sel_a, _sel_b = RESOLVERS[kind]
    root = tmp_path / kind
    out = tmp_path / f"{kind}-out"
    make_tree(root)

    mod.resolve(root, out, select=sel_a)
    text = (out / FAMILY / name).read_text()
    staged = yaml.safe_load(text)

    blocks = []
    for case in staged["cases"].values():
        for blk in (case.get("expected") or {}).values():
            blocks.append(blk)
        for ch in case.get("chunks") or []:
            for deltas in (ch.get("expected") or {}).values():
                blocks.append(deltas)
    assert len(blocks) == 2, f"{kind}: expected 2 shared blocks, got {len(blocks)}"
    assert blocks[0] is blocks[1], (
        f"{kind}: the source's shared block was expanded into two copies; the staged "
        f"file no longer matches the on-disk fold's output:\n{text}"
    )
