#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared reading + version-ordering helpers for the three fixture resolvers.

resolve_fixtures.py (batch), resolve_stream_fixtures.py (stream) and
resolve_reasoning_fixtures.py all walk the same corpus layout —

    <root>/inputs/<family>/*.yaml            shared, version-independent inputs
    <root>/<impl>-<version>/<family>/*.yaml  per-impl overlays, lowest = full anchor

— and each had grown its own copy of `load`, `version_key` and the "<impl>-<version>"
splitter. They live here once so a corpus-layout change lands in one place.
"""
import re
from pathlib import Path

import yaml
import yaml_fast  # noqa: F401 — routes safe_load/safe_dump through libyaml


def load(p):
    return yaml.safe_load(Path(p).read_text())


def version_key(ver: str):
    """Order releases, post releases, then append-only patches of each release."""
    patch_match = re.fullmatch(r"(.+)\.patch(\d+)", ver)
    version = patch_match.group(1) if patch_match else ver
    patch = int(patch_match.group(2)) if patch_match else 0
    m = re.match(r"(\d+(?:\.\d+)*)(?:[.-]?post(\d+))?", version)
    release = tuple(int(x) for x in m.group(1).split(".")) if m else ()
    post = int(m.group(2)) if m and m.group(2) else 0
    return (release, post, patch)


def split_sel(sel: str):
    """'vllm_python-0.24.0' -> ('vllm_python', '0.24.0'). An impl key may contain '_',
    but the version token always starts after the FIRST '-'."""
    impl, _, ver = sel.partition("-")
    return impl, ver


class FixtureCorpus(dict):
    """Parsed documents with discovery metadata owned by the same snapshot."""

    def __init__(self, root, documents):
        super().__init__(documents)
        self.by_directory = {}
        for (top, family, name), doc in self.items():
            self.by_directory.setdefault(top, {})[(family, name)] = doc
        self.version_dirs = {}
        self.families_by_directory = {}
        for directory in Path(root).iterdir():
            if not directory.is_dir() or directory.name == "inputs" or "-" not in directory.name:
                continue
            impl, version = split_sel(directory.name)
            self.version_dirs.setdefault(impl, []).append((version_key(version), version, directory))
            self.families_by_directory[directory.name] = {
                path.name for path in directory.iterdir() if path.is_dir()
            }
        for versions in self.version_dirs.values():
            versions.sort(key=lambda item: item[0])


def index_corpus(root, corpus=None) -> FixtureCorpus:
    """Accept legacy document mappings while reusing an already indexed corpus."""
    if corpus is None:
        return load_corpus(root)
    return corpus if isinstance(corpus, FixtureCorpus) else FixtureCorpus(root, corpus)


def load_corpus(root) -> FixtureCorpus:
    """Parse every fixture under <root> ONCE: {(top_dir, family, filename): doc}.

    `top_dir` is "inputs" or an "<impl>-<version>" dir. A caller resolving many version
    selections out of the same corpus (the generator's version-status maps resolve ~11
    for stream, ~9 for batch) parses once and reuses the result, instead of re-reading
    all ~1700 source files per selection.
    """
    root = Path(root)
    return FixtureCorpus(root, {
        (fp.parent.parent.name, fp.parent.name, fp.name): yaml.safe_load(fp.read_text())
        for fp in root.glob("*/*/*.yaml")
    })
