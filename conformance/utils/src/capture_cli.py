#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture entry point (audit B3): one argparse CLI for refreshing v2 fixtures.

Replaces the per-subcommand bash option parsing in capture.sh (now a thin wrapper).
Subcommands:
  stream                 vLLM Python/Rust + SGLang per-chunk stream capture (capture_driver)
  batch-on-stream        each batch sample's text through each stream parser (capture_driver)
  dynamo-stream          record the Dynamo v2 stream parser over one fixture (cargo bin)
  dynamo-batch-on-stream record the Dynamo v2 batch-via-stream parser (cargo bin)
  token-ids              stamp token ids into stream fixtures (cargo bin)

`--dry-run` prints the commands instead of running them. `CARGO` env overrides the
cargo binary (e.g. `CARGO='cargo +1.96.1'`). `--family`/`--fixture` (B4) narrow the
peer captures to one family/fixture.
"""
import argparse
import os
import shlex
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# capture_cli.py lives in conformance/utils/src/, so the repo root is three dirs up.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
CARGO = os.environ.get("CARGO", "cargo")
DRIVER = os.path.join(HERE, "capture_driver.py")
DYNAMO_TODO = (
    "Dynamo parser v2 TC streaming not yet implemented for this family; "
    "vLLM/SGLang per-chunk output is the target to match."
)


def _run(cmd, dry, cwd=None, stdout_path=None):
    shown = " ".join(shlex.quote(c) for c in cmd)
    if stdout_path:
        shown += f" > {shlex.quote(stdout_path)}"
    if cwd:
        shown = f"(cd {shlex.quote(cwd)} && {shown})"
    if dry:
        print(f"[dry-run] {shown}")
        return
    out = open(stdout_path, "w") if stdout_path else None
    try:
        subprocess.run(cmd, cwd=cwd, stdout=out, check=True)
    finally:
        if out:
            out.close()


def _cargo_bin(bin_name, extra, dry, output=None):
    cmd = [*CARGO.split(), "run", "-p", "dynamo-parsers-v2", "--bin", bin_name, *extra]
    _run(cmd, dry, cwd=ROOT, stdout_path=output)


def _driver(mode, args, dry, extra=()):
    work = args.work or os.path.join(tempfile.gettempdir(), f"capture_{mode.replace('-', '_')}_{os.getpid()}")
    os.makedirs(work, exist_ok=True)
    cmd = ["python3", DRIVER, "--mode", mode, "--root", ROOT, "--work", work,
           "--vllm-container", args.vllm_container, "--sglang-container", args.sglang_container]
    if args.vllm_rust_source:
        cmd += ["--vllm-rust-source", args.vllm_rust_source]
    if getattr(args, "family", None):
        cmd += ["--family", args.family]
    if getattr(args, "fixture", None):
        cmd += ["--fixture", args.fixture]
    cmd += list(extra)
    _run(cmd, dry)


def main(argv=None):
    # --dry-run is accepted anywhere (before or after the subcommand), matching the
    # old capture.sh; strip it before argparse so subparsers don't reject it.
    argv = list(sys.argv[1:] if argv is None else argv)
    dry = any(a in ("--dry-run", "--dryrun") for a in argv)
    argv = [a for a in argv if a not in ("--dry-run", "--dryrun")]

    ap = argparse.ArgumentParser(prog="capture.sh", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_peer_opts(p, with_family=True):
        p.add_argument("--vllm-container", default="vllm-localdev")
        p.add_argument("--sglang-container", default="sglang-localdev")
        p.add_argument("--vllm-rust-source")
        p.add_argument("--work")
        if with_family:
            p.add_argument("--family")
            p.add_argument("--fixture")

    s = sub.add_parser("stream")
    add_peer_opts(s)
    s.add_argument("--dynamo-todo", default=DYNAMO_TODO)

    b = sub.add_parser("batch-on-stream")
    add_peer_opts(b)
    g = b.add_mutually_exclusive_group()
    # Public flag names stay --dynamo-rust-json / --capture-dynamo-rust-json for
    # CLI compat; the dests follow the dynamo_v2 impl rename.
    g.add_argument("--dynamo-rust-json", dest="dynamo_v2_json")
    g.add_argument("--capture-dynamo-rust-json", dest="capture_dynamo_v2_json")

    ds = sub.add_parser("dynamo-stream")
    ds.add_argument("--fixture", required=True)
    ds.add_argument("--text", action="store_true")
    ds.add_argument("--output")

    db = sub.add_parser("dynamo-batch-on-stream")
    db.add_argument("--output", required=True)

    sub.add_parser("token-ids")

    args = ap.parse_args(argv)

    if args.cmd == "stream":
        _driver("stream", args, dry, extra=["--dynamo-todo", args.dynamo_todo])
    elif args.cmd == "batch-on-stream":
        dynamo_v2_json = args.dynamo_v2_json
        if args.capture_dynamo_v2_json:
            _cargo_bin("record_batch_via_stream", [], dry, output=args.capture_dynamo_v2_json)
            dynamo_v2_json = args.capture_dynamo_v2_json
        extra = ["--dynamo-rust-json", dynamo_v2_json] if dynamo_v2_json else []
        _driver("batch-on-stream", args, dry, extra=extra)
    elif args.cmd == "dynamo-stream":
        extra = ["--", args.fixture] + (["--text"] if args.text else [])
        _cargo_bin("record_dynamo_stream", extra, dry, output=args.output)
    elif args.cmd == "dynamo-batch-on-stream":
        _cargo_bin("record_batch_via_stream", [], dry, output=args.output)
    elif args.cmd == "token-ids":
        _cargo_bin("stamp_stream_token_ids", [], dry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
