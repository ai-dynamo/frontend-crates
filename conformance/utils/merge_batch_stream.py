#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge the three per-engine flat stream-on-batch captures into the nested
harmony_batch_stream.json the parity generator consumes for the "Stream parser on
batch" tab.

Inputs are flat {cid: {calls: [...]}} files, one per engine:
  --dynamo  from `cargo run -p dynamo-parsers-v2 --bin record_batch_via_stream`
  --vllm    from `capture_harmony_batch_stream.py --impl vllm`   (vllm-localdev)
  --sglang  from `capture_harmony_batch_stream.py --impl sglang` (sglang-localdev)

Output is nested {cid: {dynamo: {calls}, vllm: {calls}, sglang: {calls}}}.
Union of all cids; an engine missing a cid gets an empty calls list.

Full recipe:
  cargo run -p dynamo-parsers-v2 --bin record_batch_via_stream > /tmp/dynamo_bs.json
  # in each container, with conformance harmony batch model_text/tools as input.json:
  docker exec vllm-localdev   python3 capture_harmony_batch_stream.py --impl vllm   --input input.json > /tmp/vllm_bs.json
  docker exec sglang-localdev python3 capture_harmony_batch_stream.py --impl sglang --input input.json > /tmp/sglang_bs.json
  python3 merge_batch_stream.py --dynamo /tmp/dynamo_bs.json --vllm /tmp/vllm_bs.json \
      --sglang /tmp/sglang_bs.json -o conformance/utils/harmony_batch_stream.json
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamo", required=True)
    ap.add_argument("--vllm", required=True)
    ap.add_argument("--sglang", required=True)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    layers = {
        "dynamo": json.load(open(args.dynamo)),
        "vllm": json.load(open(args.vllm)),
        "sglang": json.load(open(args.sglang)),
    }
    cids = sorted({cid for layer in layers.values() for cid in layer})
    nested = {
        cid: {
            engine: {"calls": layer.get(cid, {}).get("calls", [])}
            for engine, layer in layers.items()
        }
        for cid in cids
    }
    json.dump(nested, open(args.output, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {args.output}: {len(nested)} cases × {len(layers)} engines")


if __name__ == "__main__":
    main()
