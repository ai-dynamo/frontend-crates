#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture an engine's streaming parser over batch fixture text.

Runs inside an engine container. Each batch case's complete `model_text` is fed
as one streaming increment, then the emitted deltas are assembled into the same
{calls, normal_text} shape used by the batch-on-stream table.
"""
import argparse
import json

import yaml

from capture_harmony_batch_stream import capture_sglang as capture_harmony_sglang
from capture_harmony_batch_stream import capture_vllm as capture_harmony_vllm
from capture_stream import capture_sglang, capture_vllm, engine_version


def _assemble(chunks):
    names, args, order = {}, {}, []
    normal_parts = []
    errors = []
    for chunk in chunks:
        if chunk.get("error"):
            errors.append(chunk["error"])
        if chunk.get("normal_text"):
            normal_parts.append(chunk["normal_text"])
        for delta in chunk.get("deltas", []):
            idx = delta["index"]
            if idx not in order:
                order.append(idx)
            if delta.get("name") is not None:
                names[idx] = names.get(idx, "") + delta["name"]
            if delta.get("arguments") is not None:
                args[idx] = args.get(idx, "") + delta["arguments"]
    calls = []
    for idx in order:
        raw = args.get(idx, "")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = raw
        calls.append({"name": names.get(idx, ""), "arguments": parsed})
    block = {"calls": calls}
    normal_text = "".join(normal_parts)
    if normal_text:
        block["normal_text"] = normal_text
    if errors:
        block["error"] = "; ".join(errors)
    return block


def _batch_cases_to_stream_cases(cases):
    out = {}
    for cid, case in cases.items():
        if "model_text" not in case:
            continue
        out[cid] = {
            "tools": case.get("tools") or [],
            "chunks": [{"delta_text": case.get("model_text") or ""}],
        }
    return out


def _harmony_cases(cases):
    out = {}
    for cid, case in cases.items():
        if "model_text" not in case:
            continue
        out[cid] = {
            "model_text": case.get("model_text") or "",
            "tools": case.get("tools") or [],
        }
    return out


def capture_fixture(impl, parser, fixture):
    doc = yaml.safe_load(open(fixture))
    family = doc["family"]
    cases = doc.get("cases", {})
    if family == "harmony":
        harmony_cases = _harmony_cases(cases)
        fn = capture_harmony_vllm if impl == "vllm" else capture_harmony_sglang
        return fn(harmony_cases)

    stream_cases = _batch_cases_to_stream_cases(cases)
    fn = capture_vllm if impl == "vllm" else capture_sglang
    per_chunk = fn(parser, stream_cases)
    return {cid: _assemble(chunks) for cid, chunks in per_chunk.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, choices=("vllm", "sglang"))
    ap.add_argument("--fixture")
    ap.add_argument("--parser")
    ap.add_argument("--batch", help="JSON: [{fixture, parser}, ...]")
    args = ap.parse_args()

    if args.batch:
        jobs = json.loads(args.batch)
        fixtures = {}
        for job in jobs:
            try:
                fixtures[job["fixture"]] = {
                    "cases": capture_fixture(
                        args.impl, job["parser"], job["fixture"]
                    )
                }
            except Exception as e:
                fixtures[job["fixture"]] = {"error": str(e)}
        print(
            json.dumps(
                {"version": engine_version(args.impl), "fixtures": fixtures},
                ensure_ascii=False,
            )
        )
        return

    result = capture_fixture(args.impl, args.parser, args.fixture)
    print(
        json.dumps(
            {"version": engine_version(args.impl), "cases": result},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
