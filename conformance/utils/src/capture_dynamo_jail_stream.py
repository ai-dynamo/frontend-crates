#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture the Dynamo v1 batch parser run against STREAM data (the streaming jail),
now that the jail buffer + v1 batch parser live together in dynamo-parsers (DIS-2296).

For each family under fixtures-stream-v2/inputs/, feed the per-chunk `delta_text` to
`JailedStream` (via the record_dynamo_jail_stream bin) and write the per-chunk output
to fixtures-stream-v2/dynamo_rust-3.0.0/<family>/<file> — the v1 (jail) stream
candidate, alongside the v2 stream parser at dynamo_rust-0.1.11/. The gpt-oss token-id
row (harmony) is recorded as unavailable (the v1 parser is text-only — see the module
note below); every other family records the jail's real per-chunk output, including an
empty result when the jail drops a call (a real divergence, not n/a).

Usage: capture_dynamo_jail_stream.py [--root <repo>]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# gpt-oss has two conformance rows for the same content: `harmony` documents the
# token-id ingestion path, `harmony_text` the text path. The v1 harmony parser is
# text-only — it takes text and re-tokenizes it via the harmony encoding
# (harmony_parser.rs: `enc.tokenizer().encode_with_special_tokens(text)`); it has no
# token-id entry point. The jail buffers streamed *text*, so v1 jail+batch serves the
# text row (harmony_text) and is n/a on the token-id row (harmony).
NO_V1_JAIL = {"harmony"}
UNAVAILABLE_MSG = (
    "Dynamo v1 has no token-id ingestion path — the harmony parser is text-only "
    "(re-tokenizes text via the harmony encoding). See the harmony_text (text) row."
)
# Map a conformance family to the v1 parser name the jail should use, where they differ
# (the text row's family is `harmony_text` but the registered v1 parser is `harmony`).
PARSER_NAME = {"harmony_text": "harmony"}
# NOTE: a case where the jail emits no call is recorded as a real empty result
# ({calls: [], normal_text: ...}), NOT as unavailable. Emitting zero calls is a genuine
# parser result that may diverge from a peer, per _derive_stream_expected in fixtures.py
# ("emitting zero calls is a real result … not a 'not applicable'"). Where the v1 jail
# drops a call the v1 batch parser recovers (e.g. deepseek_v3, whose parser is sensitive
# to the streamed text's whitespace), that divergence is exactly the signal the stream
# tab should surface — masking it as n/a would hide a real jail-vs-batch discrepancy.


def dynamo_v1_version(repo: Path) -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', (repo / "parsers" / "Cargo.toml").read_text(), re.M)
    return m.group(1)


def _canon_delta_args(deltas: list) -> list:
    """Sort each tool-call argument's JSON keys in place so the fixture output is
    deterministic across captures. The v1 parser serializes arguments from a hash map,
    so key order is otherwise run-dependent (same keys, different order) and the fixtures
    churn on every re-capture. Compact separators match the parser's own style, so only
    reordered keys change; scalars and already-sorted args stay byte-identical. Fragments
    / non-JSON (intentional truncated-call test cases) are left untouched."""
    for d in deltas:
        a = d.get("arguments")
        if not isinstance(a, str):
            continue
        try:
            obj = json.loads(a)
        except (json.JSONDecodeError, ValueError):
            continue
        d["arguments"] = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return deltas


def run_bin(repo: Path, family: str, cases: dict[str, list[str]]) -> dict:
    """Run record_dynamo_jail_stream over one family's cases; returns {cid: [chunk...]}."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump({"family": family, "cases": cases}, tf)
        inp = tf.name
    try:
        out = subprocess.run(
            [os.environ.get("CARGO", "cargo"), "run", "-q", "-p", "dynamo-parsers",
             "--bin", "record_dynamo_jail_stream", "--", inp],
            check=True, capture_output=True, text=True, cwd=str(repo),
        ).stdout
    finally:
        os.unlink(inp)
    return json.loads(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[3]))
    args = ap.parse_args(argv)
    repo = Path(args.root)
    sv2 = repo / "conformance/toolcalling/fixtures-stream-v2"
    inputs = sv2 / "inputs"
    ver = dynamo_v1_version(repo)
    out_root = sv2 / f"dynamo_rust-{ver}"

    families = sorted(d.name for d in inputs.iterdir() if d.is_dir())
    for family in families:
        for fp in sorted((inputs / family).glob("*.yaml")):
            doc = yaml.safe_load(fp.read_text())
            cases = doc.get("cases") or {}
            out_cases: dict[str, dict] = {}
            if family in NO_V1_JAIL:
                for cid in cases:
                    out_cases[cid] = {"unavailable": UNAVAILABLE_MSG.format(family=family)}
            else:
                # {cid: [delta_text per chunk]} for the recorder
                bin_in = {
                    cid: [ch.get("delta_text", "") for ch in (c.get("chunks") or [])]
                    for cid, c in cases.items()
                }
                recorded = run_bin(repo, PARSER_NAME.get(family, family), bin_in)
                for cid in cases:
                    # Always record the jail's actual per-chunk output — including an
                    # empty result when it drops the call. Empty is a real result that
                    # diverges from the reference/peers, not a "not applicable" (see the
                    # module note above); marking it unavailable would hide that.
                    chunks = []
                    for ch in recorded.get(cid, []):
                        entry = {"expected": _canon_delta_args(ch.get("deltas") or [])}
                        if ch.get("normal_text"):
                            entry["normal_text"] = ch["normal_text"]
                        chunks.append(entry)
                    out_cases[cid] = {"chunks": chunks}
            dst = out_root / family / fp.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(yaml.safe_dump(
                {"family": doc.get("family", family), "mode": "streamv2",
                 "captured_with": {"dynamo_rust": ver}, "cases": out_cases},
                sort_keys=False, allow_unicode=True, width=4096,
            ))
        print(f"{family}: captured", file=sys.stderr)
    print(f"wrote {out_root} (dynamo v1 jail @ {ver})", file=sys.stderr)


if __name__ == "__main__":
    main()
