#!/usr/bin/env python3
"""Apply ONE capture to ONE impl version tree, in full.

The counterpart to write_impl_layers.py for an impl published at a single version
(vllm_rust): there is no higher version to diff against, so the tree is the full anchor
and every captured case is stored. Both writers share capture_layers.py so the stored
per-case shape (`expected` + `normal_text` per chunk) cannot drift between them.

Usage: apply_capture.py <capture.json> <tree-dir> <impl> [--version VER] [--apply]

`--version` overrides what goes into `captured_with`. The Rust probe reports a source
version ("v0.23.0 <sha>") that identifies the checkout, not the published version the
tree dir is named for; `captured_with` has to keep matching the dir name.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_layers import load_capture, write_full_tree  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("capture")
ap.add_argument("tree_dir")
ap.add_argument("impl")
ap.add_argument("--version", help="value to stamp into captured_with[<impl>]")
ap.add_argument("--apply", action="store_true")
args = ap.parse_args()
cap_path, tree_dir, impl, APPLY = args.capture, args.tree_dir, args.impl, args.apply

captured_version, cases = load_capture(cap_path)
version = args.version or captured_version
print(f"{impl}: capture={captured_version} -> captured_with={version} "
      f"({len(cases)} cases) -> {tree_dir}")
written = write_full_tree(tree_dir, cases, impl, version, apply=APPLY)
print(f"  cases written: {written}{'' if APPLY else '  [dry run, pass --apply]'}")
