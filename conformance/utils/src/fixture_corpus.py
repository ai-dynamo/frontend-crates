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
import copy
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


def complete_family_snapshots(root: Path) -> bool:
    """Read the materializer contract, including readers nested under .reader-views."""
    root = Path(root).resolve()
    for parent in (root, *root.parents):
        marker = parent / ".reader-views/legacy-checkpoints.json"
        if marker.exists():
            metadata = load(marker)
            if metadata != {"schema_version": 1, "complete_family_snapshots": True}:
                raise ValueError(f"invalid materialized checkpoint marker: {marker}")
            return True
    return False


def checkpoint_families(directory: Path) -> set[str]:
    """An empty family directory is an explicitly measured empty checkpoint."""
    return {path.name for path in directory.iterdir() if path.is_dir()}


def clear_checkpoint_observations(root: Path, directory: Path, docs: dict, impl: str, *, corpus: dict, stream: bool = False, introduced_unavailable: set | None = None) -> set:
    """Replace a complete family's implementation state before merging its cases.

    Shared requests and other implementations remain intact. Older sparse layouts
    retain overlay behavior, and families absent from this checkpoint still inherit.
    """
    if not complete_family_snapshots(root):
        return set()
    families = checkpoint_families(directory)
    touched = set()
    for key, doc in docs.items():
        if key[0] not in families:
            continue
        touched.add(key)
        doc.get("captured_with", {}).pop(impl, None)
        input_cases = corpus.get(("inputs", *key), {}).get("cases", {})
        for case_id, case in doc.get("cases", {}).items():
            original = input_cases.get(case_id, {})
            had_unavailable = "unavailable" in case
            for field in ("expected", "unavailable", "exception"):
                if isinstance(case.get(field), dict):
                    case[field].pop(impl, None)
            for chunk in case.get("chunks", []):
                if not isinstance(chunk, dict):
                    continue
                for field in ("expected", "normal_text"):
                    if isinstance(chunk.get(field), dict):
                        chunk[field].pop(impl, None)
            reason = original.get("unavailable", {}).get(impl, "No observation was recorded for this case in the selected capture.")
            if impl in original.get("unavailable", {}):
                case.setdefault("unavailable", {})[impl] = copy.deepcopy(reason)
            if not any(field in original for field in ("model_text", "input", "chunks", "expected", "unavailable")):
                continue
            if stream:
                case.setdefault("unavailable", {})[impl] = copy.deepcopy(reason)
                if not had_unavailable and introduced_unavailable is not None:
                    introduced_unavailable.add((*key, case_id))
            else:
                recorded = original.get("expected", {}).get(impl, {})
                case.setdefault("expected", {})[impl] = (copy.deepcopy(recorded) if "unavailable" in recorded
                                                         else {"unavailable": copy.deepcopy(reason)})
    return touched


def cleanup_checkpoint_observations(docs: dict, introduced_unavailable: set) -> None:
    """Track synthetic wrappers until a later checkpoint replaces their missing state."""
    for family, filename, case_id in tuple(introduced_unavailable):
        case = docs[(family, filename)]["cases"][case_id]
        if case.get("unavailable") == {}:
            del case["unavailable"]
        if "unavailable" not in case:
            introduced_unavailable.discard((family, filename, case_id))
