#!/usr/bin/env python3
"""Write a peer impl's anchor + changed-only overlay trees from two captures.

Corpus layout: the LOWEST published version of an impl is the full anchor; every higher
version stores only the cases whose output DIFFERS from the anchor. Re-capturing both
versions and diffing is the only way to keep that invariant true — updating just the
cases an overlay already happens to contain silently keeps stale entries and drops cases
that newly diverged.

The stored per-case shape (`expected` + `normal_text` per chunk) and the anchor-driven
overlay rebuild both live in capture_layers.py, shared with apply_capture.py so the two
writers cannot drift.

Usage: write_impl_layers.py <anchor.json> <anchor-dir> <over.json> <over-dir> <impl> [--apply]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_layers import load_capture, write_full_tree, write_overlay_tree  # noqa: E402

anchor_cap, anchor_dir, over_cap, over_dir, impl = sys.argv[1:6]
APPLY = "--apply" in sys.argv

av, anchor = load_capture(anchor_cap)
ov, over = load_capture(over_cap)
print(f"{impl}: anchor={av} ({len(anchor)} cases)  overlay={ov} ({len(over)} cases)")

n_anchor = write_full_tree(anchor_dir, anchor, impl, av, apply=APPLY)
n_overlay, n_removed = write_overlay_tree(over_dir, over, anchor_dir, anchor, impl, ov,
                                          apply=APPLY)
print(f"  cases written: anchor={n_anchor}  overlay(changed-only)={n_overlay}"
      f"  overlay files removed (no changed case): {n_removed}"
      f"{'' if APPLY else '  [dry run, pass --apply]'}")
