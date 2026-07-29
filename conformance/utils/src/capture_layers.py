#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one place that turns a capture envelope into a per-impl version tree.

Two writers need this: write_impl_layers.py (anchor + changed-only overlay, for an
impl published at two versions) and apply_capture.py (a single full tree, for an impl
published at one). They MUST store the same per-case body, so the shape lives here
once instead of being copy-pasted into both.

Per chunk the tree stores `expected` (the deltas) and, when the capture recorded
non-tool text for that chunk, `normal_text`. Dropping `normal_text` is not cosmetic:
it is the only field markers._block_tool_call_leaks reads, so a tree written without
it reports zero tool-call-markup leaks for that impl no matter what the parser
actually emitted.
"""
import pathlib
import json

import yaml

STREAM_GLOB = "*/TOOLCALLING.streamv2*.yaml"


def load_capture(path):
    """(version, {(family, filename, case_id): result}) from a capture envelope.

    The envelope is what recapture_peer.py and capture_vllm_rust.py --batch both emit:
    {"version": str, "fixtures": {fixture_path: {"cases": {case_id: result}}}}.
    """
    cap = json.load(open(path))
    out = {}
    for fx, body in cap["fixtures"].items():
        p = pathlib.Path(fx)
        for cid, res in (body.get("cases") or {}).items():
            out[(p.parent.name, p.name, cid)] = res
    return cap["version"], out


def case_body(res, impl):
    """Capture result -> the fields a version tree stores for one case."""
    if isinstance(res, dict) and "error" in res:
        return {"unavailable": f"{impl} parser not captured: {res['error']}"}
    chunks = []
    for ch in res:
        entry = {"expected": ch.get("deltas") or []}
        # Only non-empty normal_text is stored, matching the rest of the corpus:
        # resolve_stream_fixtures._merge_impl reads a falsy value as "this impl emitted
        # no normal text here" and drops the key from the resolved chunk anyway.
        normal_text = ch.get("normal_text")
        if normal_text:
            entry["normal_text"] = normal_text
        chunks.append(entry)
    return {"chunks": chunks}


def _dump(doc, path, apply):
    out = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=4096)
    assert yaml.safe_load(out) == doc, f"{path}: round-trip mismatch"
    if apply:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(out)


def write_full_tree(tree, cap, impl, version, apply=False):
    """Rewrite an impl's FULL tree (an anchor, or a single-version impl's only tree).

    Cases the capture does not cover keep whatever the tree already held — that is how
    a hand-authored `unavailable` note for a family this impl has no parser for
    survives a re-capture. Returns the number of cases written from the capture.
    """
    root = pathlib.Path(tree)
    written = 0
    for f in sorted(root.glob(STREAM_GLOB)):
        doc = yaml.safe_load(f.read_text()) or {}
        keep = {}
        for cid, prior in (doc.get("cases") or {}).items():
            res = cap.get((f.parent.name, f.name, cid))
            if res is None:
                keep[cid] = prior                     # impl not applicable here
                continue
            keep[cid] = case_body(res, impl)
            written += 1
        doc["cases"] = keep
        # `captured_with` names the version this tree speaks for.
        doc.setdefault("captured_with", {})[impl] = version
        _dump(doc, f, apply)
    return written


def write_overlay_tree(over_dir, over_cap, anchor_dir, anchor_cap, impl, version,
                       apply=False):
    """Rebuild an impl's changed-only overlay from the ANCHOR's case universe.

    The overlay must be driven by the anchor tree, not by the cases the overlay file
    already happens to list: a case that diverges only in `normal_text` was absent from
    the overlay before this fix, and reading the overlay's own case list would make that
    absence permanent — the overlay could only ever shrink. Iterating the anchor lets a
    newly divergent case reappear.

    A case the capture does not cover carries the existing overlay entry through
    verbatim (hand-authored `unavailable` notes). An overlay file that ends up with no
    cases is removed, matching the corpus convention that every overlay file holds at
    least one changed case.
    """
    over_root, anchor_root = pathlib.Path(over_dir), pathlib.Path(anchor_dir)
    written, removed = 0, 0
    for af in sorted(anchor_root.glob(STREAM_GLOB)):
        anchor_doc = yaml.safe_load(af.read_text()) or {}
        of = over_root / af.parent.name / af.name
        over_doc = yaml.safe_load(of.read_text()) if of.exists() else None
        prior = (over_doc or {}).get("cases") or {}
        keep = {}
        # Anchor order first, then any overlay-only case ids, so a case the anchor
        # does not carry still gets its existing entry preserved.
        for cid in list(anchor_doc.get("cases") or {}) + [
            c for c in prior if c not in (anchor_doc.get("cases") or {})
        ]:
            key = (af.parent.name, af.name, cid)
            res, base = over_cap.get(key), anchor_cap.get(key)
            if res is None or base is None:
                if cid in prior:
                    keep[cid] = prior[cid]            # not captured -> carry through
                continue
            body = case_body(res, impl)
            if body == case_body(base, impl):
                continue                              # identical to anchor -> omit
            keep[cid] = body
            written += 1
        if not keep:
            if of.exists():
                removed += 1
                if apply:
                    of.unlink()
            continue
        doc = over_doc or {"family": anchor_doc.get("family"),
                           "mode": anchor_doc.get("mode")}
        doc["cases"] = keep
        doc.setdefault("captured_with", {})[impl] = version
        # Reorder so a newly created file matches the corpus header order.
        doc = {k: doc[k] for k in ("family", "mode", "captured_with", "cases")
               if k in doc} | {k: v for k, v in doc.items()
                               if k not in ("family", "mode", "captured_with", "cases")}
        _dump(doc, of, apply)
    if apply:
        for d in sorted(over_root.glob("*")):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    return written, removed
