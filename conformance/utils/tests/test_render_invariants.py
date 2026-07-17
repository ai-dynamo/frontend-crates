# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-fold contract on the resolver (pure logic, no rendered HTML).

Born from a real escape: the resolver fold doubled Dynamo v1 output into
`calls=[get_weatherget_weather(...)]` and it shipped to the rendered page unnoticed
because verification stopped at "the data exists", never "the output is right".

DIS-2434: the page is now rendered by the JS view from the JSON model, so the former
HTML-regex guards here (grid↔compare-bar keys, versions-latest-first, no doubled
names, chart-vs-list exclusivity, v2 registry cross-check) moved to structural
assertions on the model (test_model.py) and DOM smokes (test_browser_popup_compare.py)
— stronger and less brittle. What remains is the one guard with no HTML at all: folding
a higher Dynamo version reproduces that version's docs exactly (the fold contract the
rendered stream cells depend on).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

UTILS = Path(__file__).resolve().parents[1]
SRC = UTILS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resolve_stream_fixtures import resolve, version_key  # noqa: E402

FIXTURES_ROOT = Path(
    os.environ.get(
        "CONFORMANCE_FIXTURES_ROOT",
        os.path.expanduser("~/.cache/dynamo/conformance-fixtures"),
    )
)
STREAM_SRC = FIXTURES_ROOT / "toolcalling" / "fixtures-stream-v2"

pytestmark = pytest.mark.skipif(
    not STREAM_SRC.is_dir(), reason="conformance fixtures not downloaded"
)


def _dynamo_version_dirs() -> list[tuple[str, Path]]:
    out = []
    for d in STREAM_SRC.iterdir():
        if d.is_dir() and d.name.startswith("dynamo_v2-"):
            out.append((d.name.split("-", 1)[1], d))
    out.sort(key=lambda t: version_key(t[0]))
    return out


# --- folding a higher dynamo version reproduces that version's docs exactly ----
def test_fold_reproduces_each_dynamo_version_exactly():
    versions = _dynamo_version_dirs()
    if len(versions) < 2:
        pytest.skip("needs at least two dynamo_v2 version dirs")
    top_ver, top_dir = versions[-1]
    with tempfile.TemporaryDirectory() as tmp:
        resolve(STREAM_SRC, tmp, select=[f"dynamo_v2-{top_ver}"])
        for vfp in top_dir.glob("*/*.yaml"):
            folded_fp = Path(tmp) / vfp.parent.name / vfp.name
            if not folded_fp.exists():
                continue
            want_doc = yaml.safe_load(vfp.read_text()) or {}
            got_doc = yaml.safe_load(folded_fp.read_text()) or {}
            for cid, want_case in (want_doc.get("cases") or {}).items():
                got_case = (got_doc.get("cases") or {}).get(cid)
                if got_case is None or "unavailable" in want_case:
                    continue
                got = [
                    (i, d)
                    for i, ch in enumerate(got_case.get("chunks") or [])
                    for d in ((ch.get("expected") or {}).get("dynamo_v2") or [])
                ]
                want = [
                    (i, d)
                    for i, ch in enumerate(want_case.get("chunks") or [])
                    for d in (ch.get("expected") or [])
                ]
                assert got == want, (
                    f"{vfp.parent.name}/{vfp.name} {cid}: folded dynamo_v2 deltas "
                    f"differ from the {top_ver} doc (lower-version residue?)\n"
                    f"  got:  {got}\n  want: {want}"
                )
