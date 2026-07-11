#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture the Dynamo v1 streaming JAIL output over the streamv2 chunk corpus.

Mirrors capture_vllm_rust.py's source-path model: the jail lives in the *dynamo*
repo (lib/llm), so we run it from a checked-out dynamo source tree passed via
`--dynamo-source` (or DYN_SOURCE). The recorder is the in-crate test
`record_jail_streamv2` (the jail's per-chunk `process_content` is crate-internal,
so it can't be a `src/bin`). It reads DYN_STREAMV2_IN and writes DYN_JAIL_OUT.

Usage:
  capture_dynamo_jail.py --dynamo-source ~/dynamo/dynamoX \
      --streamv2-in /tmp/streamv2_inputs.json --out /tmp/jail.json

Output JSON: {family: {case: [ {deltas:[{index,name?,arguments?}], normal_text}, ... per chunk ]}}
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

CARGO = os.environ.get("CARGO", "cargo")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamo-source", default=os.environ.get("DYN_SOURCE"),
                    help="dynamo checkout root that carries the record_jail_streamv2 recorder")
    ap.add_argument("--streamv2-in", required=True,
                    help="JSON of streamv2 inputs: [{family, case, chunks:[delta_text], tools}]")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if not args.dynamo_source:
        sys.exit("error: --dynamo-source (or DYN_SOURCE) is required")
    src = os.path.abspath(os.path.expanduser(args.dynamo_source))
    if not os.path.isdir(src):
        sys.exit(f"error: dynamo source not found: {src}")

    out_tmp = tempfile.mktemp(suffix=".json")
    env = {**os.environ, "DYN_STREAMV2_IN": os.path.abspath(args.streamv2_in),
           "DYN_JAIL_OUT": out_tmp}
    cmd = [*CARGO.split(), "test", "-p", "dynamo-llm", "--lib",
           "record_jail_streamv2", "--", "--nocapture"]
    proc = subprocess.run(cmd, cwd=src, env=env, capture_output=True, text=True)
    if proc.returncode or not os.path.exists(out_tmp):
        sys.exit(f"jail capture failed:\n{proc.stderr[-2000:] or proc.stdout[-2000:]}")
    shutil.move(out_tmp, args.out)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
