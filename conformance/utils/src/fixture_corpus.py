#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared corpus-reading and selector helpers for the three fixture resolvers.

resolve_fixtures.py (batch), resolve_stream_fixtures.py (stream) and
resolve_reasoning_fixtures.py all walk the same corpus layout —

    <root>/inputs/<family>/*.yaml            shared, version-independent inputs
    <root>/<impl>-<version>/<family>/*.yaml  per-impl overlays, lowest = full anchor

— and each had grown its own copy of `load`, `version_key` and the "<impl>-<version>"
splitter. The batch, stream, and reasoning resolvers and `unified_history.py` use
`fixture_disposition.version_sort_key` for the same release and prerelease ordering.
"""
from pathlib import Path

import yaml
import yaml_fast  # noqa: F401 — routes safe_load/safe_dump through libyaml
from fixture_disposition import version_sort_key as version_key


def load(p):
    return yaml.safe_load(Path(p).read_text())


def split_sel(sel: str):
    """'vllm_python-0.24.0' -> ('vllm_python', '0.24.0'). An impl key may contain '_',
    but the version token always starts after the FIRST '-'."""
    impl, _, ver = sel.partition("-")
    return impl, ver


def load_corpus(root) -> dict[tuple[str, str, str], dict]:
    """Parse every fixture under <root> ONCE: {(top_dir, family, filename): doc}.

    `top_dir` is "inputs" or an "<impl>-<version>" dir. A caller resolving many version
    selections out of the same corpus (the generator's version-status maps resolve ~11
    for stream, ~9 for batch) parses once and reuses the result, instead of re-reading
    all ~1700 source files per selection.
    """
    root = Path(root)
    return {
        (fp.parent.parent.name, fp.parent.name, fp.name): yaml.safe_load(fp.read_text())
        for fp in root.glob("*/*/*.yaml")
    }
