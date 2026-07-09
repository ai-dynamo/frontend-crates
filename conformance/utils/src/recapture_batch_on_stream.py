# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-capture the vLLM Python / SGLang Python STREAMING parsers over the v1 batch
text at the CURRENT engine containers, and update the batch-on-stream fixtures
in place.

The batch-on-stream tab (`Tool Calling (batch data)` on CONFORMANCE_v2) compares
each engine's streaming parser output on the batch text against Dynamo parser v2.
That page is the current-era view, so its stream candidates must show the current
engine versions. The captured fixtures were frozen at vLLM 0.23.0 / SGLang
0.5.12.post1 (see each fixture's `captured_with`), so the tab showed the stale
`vLLM Python 0.23.0 (stream)` / `SGLang Python 0.5.12.post1 (stream)`.

This re-captures ONLY the two container engines and rewrites their per-case blocks
plus their `captured_with` stamps, leaving the `vllm_rust` and `dynamo_rust` blocks
(and their `captured_with`) untouched — those are not container captures and did
not change. Run against the running dev containers:

    python3 recapture_batch_on_stream.py \
        --batch-v1 conformance/toolcalling/fixtures-batch-v1/inputs \
        --out conformance/toolcalling/fixtures-batch-on-stream-v2

Then republish the batch-on-stream tree and bump the manifest.
"""
import argparse
import glob
import os
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from capture_driver import (  # noqa: E402
    _block_for,
    _container_capture,
    _copy_worker,
    _cpath,
    _parser_for,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-v1", required=True,
                    help="v1 batch fixtures dir holding the model_text source "
                         "(the versioned corpus' inputs/ tree)")
    ap.add_argument("--out", required=True,
                    help="batch-on-stream fixtures dir to update in place")
    ap.add_argument("--vllm-container", default="vllm-localdev")
    ap.add_argument("--sglang-container", default="sglang-localdev")
    ap.add_argument("--work", default="/tmp")
    ap.add_argument("--family", help="restrict to one family (debugging)")
    args = ap.parse_args()

    _copy_worker((args.vllm_container, args.sglang_container))

    sources = sorted(glob.glob(f"{args.batch_v1}/*/TOOLCALLING.batch*.yaml"))
    if args.family:
        sources = [s for s in sources
                   if os.path.basename(os.path.dirname(s)) == args.family]

    # One batched capture per engine (single engine import) over the batch text.
    jobs: dict[str, list[dict]] = {"vllm": [], "sglang": []}
    for src in sources:
        family = os.path.basename(os.path.dirname(src))
        cpath = _cpath(src, "batch-on-stream")
        for impl in ("vllm", "sglang"):
            parser = _parser_for(impl, family)
            if parser:
                jobs[impl].append({"src": src, "container_path": cpath, "parser": parser})

    print(f"capturing vllm ({len(jobs['vllm'])} fixtures)...", file=sys.stderr)
    vllm_ver, vllm_caps = _container_capture(
        args.vllm_container, "vllm", "batch-on-stream", jobs["vllm"], args.work)
    print(f"capturing sglang ({len(jobs['sglang'])} fixtures)...", file=sys.stderr)
    sglang_ver, sglang_caps = _container_capture(
        args.sglang_container, "sglang", "batch-on-stream", jobs["sglang"], args.work)
    print(f"  vllm_python={vllm_ver}  sglang_python={sglang_ver}", file=sys.stderr)

    updated = 0
    for src in sources:
        family = os.path.basename(os.path.dirname(src))
        outfp = os.path.join(args.out, family, os.path.basename(src))
        if not os.path.exists(outfp):
            continue
        doc = yaml.safe_load(open(outfp)) or {}
        cw = doc.setdefault("captured_with", {})
        cw["vllm_python"] = vllm_ver
        cw["sglang_python"] = sglang_ver

        vllm_parser = _parser_for("vllm", family)
        sglang_parser = _parser_for("sglang", family)
        vllm_cases = _block_for("vllm_python", family, vllm_parser, vllm_caps.get(src, {}))
        sglang_cases = _block_for("sglang_python", family, sglang_parser, sglang_caps.get(src, {}))

        for cid, row in (doc.get("cases") or {}).items():
            for impl, parser, cases in (
                ("vllm_python", vllm_parser, vllm_cases),
                ("sglang_python", sglang_parser, sglang_cases),
            ):
                engine = "vLLM Python" if impl == "vllm_python" else "SGLang"
                if parser is None:
                    row[impl] = {"unavailable": f"No {engine} parser for family '{family}'."}
                elif cid in cases:
                    row[impl] = cases[cid]
                else:
                    # Preserve an existing block if the capture didn't return this
                    # case (e.g. no batch model_text); only mark unavailable if there
                    # was nothing there before.
                    if impl not in row:
                        row[impl] = {"unavailable": "Capture did not return this case."}

        with open(outfp, "w") as f:
            yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
        updated += 1
        print(f"  wrote {family}/{os.path.basename(src)}", file=sys.stderr)

    print(f"updated {updated} batch-on-stream fixtures", file=sys.stderr)


if __name__ == "__main__":
    main()
