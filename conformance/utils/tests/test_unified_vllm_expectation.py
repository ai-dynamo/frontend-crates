# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A vLLM cell nobody can capture must say so.

`unified_render.rs` reads `expect.vllm` only as a FALLBACK for a case vLLM did not
capture. The packaged feed is rendered in `conformance/CONFORMANCE_v2.html`.
For a family `capture_vllm_unified.py` cannot run at all, that fallback is every case, so
the whole column is authored — and a bare `match` there draws the same
`expected: MATCH` a family an engine really parsed earns.

This file guards the AUTHORED spec only. `expect.*` never reaches the packaged shards,
so the published `CONFORMANCE_v2.html` Unified tab cannot read this note at all; it
derives its own from the missing capture. `test_unified_tab_vllm_caveat.py` guards that
side, and neither guard implies the other.

The gap fails SILENTLY, in the direction that reads as good news: nothing errors,
the cell is simply greener than the corpus can support.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gen_unified_golden as G  # noqa: E402


def _vllm_capture_families() -> set[str]:
    """`capture_vllm_unified.FAMILY_PARSERS`, read rather than imported.

    The harness runs INSIDE a vLLM container and imports `vllm` at module scope, so
    a no-engine gate can only read it.
    """
    tree = ast.parse((SRC / "capture_vllm_unified.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "FAMILY_PARSERS" for t in node.targets
        ):
            return {k.value for k in node.value.keys}
    raise AssertionError("capture_vllm_unified.py declares no FAMILY_PARSERS")


def test_the_uncapturable_set_is_what_the_vllm_harness_cannot_run():
    # Derived, not a second opinion: adding a family to the capture harness has to
    # retire its caveat here, and dropping one has to raise it.
    assert set(G.VLLM_UNCAPTURABLE) == set(G.FAMILIES) - _vllm_capture_families()


@pytest.mark.parametrize("fam", sorted(G.VLLM_UNCAPTURABLE))
def test_every_uncapturable_vllm_cell_carries_the_caveat(fam):
    bare = [
        cid for cid, c in G.build_cases(fam).items() if not c["expect"]["vllm"].get("note")
    ]
    assert not bare


@pytest.mark.parametrize("fam", sorted(set(_vllm_capture_families()) & set(G.FAMILIES)))
def test_a_capturable_family_keeps_its_bare_verdicts(fam):
    # The mirror. A caveat that spread to a family vLLM really parses would hide the
    # one thing the annotation exists to mark, so the fix must stay this narrow.
    cases = G.build_cases(fam)
    assert any(not c["expect"]["vllm"].get("note") for c in cases.values())
