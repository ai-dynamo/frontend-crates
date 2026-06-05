#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an engine's HARMONY STREAMING parser over each batch sample's full text and
emit the assembled tool calls. This is the engine-side input for the "Stream parser
on batch" tab: does each engine's streaming parser, fed the complete batch output,
land on the same calls as its batch parser?

Runs INSIDE the engine container (vllm-localdev / sglang-localdev).

  vllm   — tokenize model_text via the harmony encoding and drive
           extract_harmony_streaming_delta one token at a time (mirrors the vLLM
           serving loop; same path as capture_vllm_harmony.py).
  sglang — drive the gpt-oss FunctionCallParser detector, feeding the full
           model_text as one streaming increment.

Input  (--input FILE): {cid: {model_text, tools: [{name, parameters}]}} JSON.
Output (stdout):       {cid: {calls: [{name, arguments}]}} JSON. arguments is
                       parsed JSON when valid, else the raw string.
The openai_harmony / engine version is printed to stderr for the record.
"""
import argparse
import importlib.metadata as meta
import json
import sys


def _assemble(deltas):
    """Fold per-index {name?, arguments?} deltas into [{name, arguments}]."""
    names, args, order = {}, {}, []
    for d in deltas:
        idx = d["index"]
        if idx not in order:
            order.append(idx)
        if d.get("name") is not None:
            names[idx] = names.get(idx, "") + d["name"]
        if d.get("arguments") is not None:
            args[idx] = args.get(idx, "") + d["arguments"]
    calls = []
    for idx in order:
        raw = args.get(idx, "")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = raw
        calls.append({"name": names.get(idx, ""), "arguments": parsed})
    return calls


def capture_vllm(cases):
    from openai_harmony import HarmonyEncodingName, StreamableParser, load_harmony_encoding
    from vllm.entrypoints.openai.chat_completion.stream_harmony import (
        TokenState,
        extract_harmony_streaming_delta,
    )

    START_TOKEN = 200006
    PREAMBLE = [200006, 173781]  # <|start|>assistant
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    out = {}
    for cid, case in cases.items():
        ids = enc.encode(case["model_text"], allowed_special="all")
        if ids and ids[0] != START_TOKEN:
            ids = PREAMBLE + ids
        parser = StreamableParser(enc, role=None)
        deltas, broken = [], False
        for tid in ids:
            if broken:
                break
            prev_recipient = parser.current_recipient
            try:
                parser.process(tid)
            except Exception:
                # Stray/terminal token after a message closes — mirror the local
                # parser, which breaks and keeps what it emitted.
                break
            ts = [
                TokenState(
                    parser.current_channel,
                    parser.current_recipient,
                    parser.last_content_delta or "",
                )
            ]
            dm, _ = extract_harmony_streaming_delta(parser, ts, prev_recipient, False)
            if dm is None:
                continue
            for tc in dm.tool_calls or []:
                d = {"index": tc.index}
                fn = tc.function
                if fn is not None:
                    if fn.name is not None:
                        d["name"] = fn.name
                    if fn.arguments is not None:
                        d["arguments"] = fn.arguments
                deltas.append(d)
        out[cid] = {"calls": _assemble(deltas)}
    print(f"openai_harmony {meta.version('openai_harmony')}", file=sys.stderr)
    return out


def capture_sglang(cases):
    sys.path.insert(0, "/sgl-workspace/sglang/python")
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.entrypoints.openai.protocol import Function, Tool

    detector_cls = FunctionCallParser.ToolCallParserEnum["gpt-oss"]
    out = {}
    for cid, case in cases.items():
        tools = [
            Tool(
                type="function",
                function=Function(name=t["name"], parameters=t.get("parameters", {})),
            )
            for t in (case.get("tools") or [])
        ]
        det = detector_cls()
        # Feed the complete batch text as a single streaming increment.
        deltas = []
        r = det.parse_streaming_increment(case["model_text"], tools)
        for c in r.calls or []:
            d = {"index": c.tool_index}
            if c.name:
                d["name"] = c.name
            if c.parameters:
                d["arguments"] = c.parameters
            deltas.append(d)
        out[cid] = {"calls": _assemble(deltas), "normal_text": r.normal_text or ""}
    print(f"sglang {meta.version('sglang')}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=("vllm", "sglang"), required=True)
    ap.add_argument("--input", required=True, help="{cid:{model_text,tools}} JSON")
    args = ap.parse_args()
    cases = json.load(open(args.input))
    out = capture_vllm(cases) if args.impl == "vllm" else capture_sglang(cases)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
