# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grammar registry and implementation aliases used by authoring/storage checks."""
import re
from pathlib import Path
from typing import Any

import yaml

from impls import LEGACY_IMPL_ALIASES

def _canonical_impl_key(impl: str) -> str:
    return LEGACY_IMPL_ALIASES.get(impl, impl)


def _legacy_impl_keys(impl: str) -> list[str]:
    return [old for old, new in LEGACY_IMPL_ALIASES.items() if new == impl]


def _impl_get(mapping: object, impl: str, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    if impl in mapping:
        return mapping[impl]
    for legacy in _legacy_impl_keys(impl):
        if legacy in mapping:
            return mapping[legacy]
    return default


def _normalize_impl_mapping(mapping: object) -> dict:
    if not isinstance(mapping, dict):
        return {}
    normalized: dict = {}
    for key, value in mapping.items():
        canonical = _canonical_impl_key(str(key))
        if canonical not in normalized:
            normalized[canonical] = value
    return normalized


# Tool-call markup leak detector (the `↯` marker): derived from the union of every
# family's declared grammar tokens in parser_families.yaml `markers:` (DIS-2442) —
# registering a family's tokens there is the ONLY step to make its leaks detectable.
#
# Truncation cases (taxonomy group 5) leak PARTIAL markers (`<tool_call`,
# `[/TOOL_CALLS`, `<｜tool▁sep｜`), so each pair/singleton token also matches by any
# prefix of at least _LEAK_PREFIX_MIN chars — the stem semantics the old hardcoded
# regex had (`</?tool_call`, `TOOL_CALLS`, ...). Tokens shorter than the minimum
# match whole only (short stems are false-positive-prone). `leak:` entries are
# already stems of parameterized tags (e.g. "<invoke" for `<invoke name="x">`) and
# are matched verbatim. Substring semantics (no anchors), longest-first alternation.
_LEAK_PREFIX_MIN = 7


def parser_families_path() -> Path:
    """The registry lives at src/parser_families.yaml in the repo; the render stage
    copies this module flat into tests/parity/ and the registry to <stage>/src/.

    Public because every consumer of the registry has to resolve BOTH layouts, and a
    second copy of this walk is a second thing to get wrong -- `unified_taxonomy` did
    exactly that and broke the render, which only runs from the stage."""
    here = Path(__file__).resolve()
    for cand in (
        here.parent / "parser_families.yaml",
        here.parents[2] / "src" / "parser_families.yaml",
    ):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"parser_families.yaml not found relative to {here}")


def _declared_leak_patterns() -> list[str]:
    spec = yaml.safe_load(parser_families_path().read_text())
    patterns: set[str] = set()
    for decl in (spec.get("markers") or {}).values():
        tokens = [t for pair in decl.get("pairs") or [] for t in pair]
        tokens += decl.get("singletons") or []
        for tok in tokens:
            patterns.update(tok[:n] for n in range(_LEAK_PREFIX_MIN, len(tok) + 1))
            patterns.add(tok)
        patterns.update(decl.get("leak") or [])
    if not patterns:
        raise ValueError("parser_families.yaml declares no family markers")
    return sorted(patterns, key=len, reverse=True)


_TOOL_CALL_MARKUP_RE = re.compile(
    "|".join(re.escape(p) for p in _declared_leak_patterns())
)
