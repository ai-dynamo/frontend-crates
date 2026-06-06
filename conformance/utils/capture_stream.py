#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture per-chunk streaming deltas from vLLM or SGLang. Runs inside the engine
container. Emits {version, cases: {cid: [{deltas, normal_text}, ...]}} as JSON.

A delta: {index, id?, name?, arguments?}. id present (truthy) => an id was emitted.

Usage (in container):
  python3 capture_stream.py --impl vllm   --fixture <stream.yaml> --parser <name>
  python3 capture_stream.py --impl sglang --fixture <stream.yaml> --parser <name>
"""
import argparse
import json
import sys

import yaml


_SPECIAL_TOKEN_IDS = {
    "[TOOL_CALLS]": 100,
    "<｜tool▁calls▁begin｜>": 101,
    "<｜tool▁calls▁end｜>": 102,
    "<｜tool▁call▁begin｜>": 103,
    "<｜tool▁call▁end｜>": 104,
    "<|python_tag|>": 105,
    "<|tool_call>": 106,
    "<tool_call|>": 107,
    "<tool_calls>": 108,
    "</tool_calls>": 109,
    "<minimax:tool_call>": 110,
    "</minimax:tool_call>": 111,
    "<tool_call>": 112,
    "</tool_call>": 113,
    '<|"|>': 114,
}
_SPECIAL_TOKENS_BY_LENGTH = sorted(_SPECIAL_TOKEN_IDS, key=len, reverse=True)


def _synthetic_token_ids(text):
    ids = []
    pos = 0
    while pos < len(text):
        matches = [
            (text.find(token, pos), token)
            for token in _SPECIAL_TOKENS_BY_LENGTH
            if text.find(token, pos) != -1
        ]
        if not matches:
            break
        start, token = min(matches)
        ids.append(_SPECIAL_TOKEN_IDS[token])
        pos = start + len(token)
    return ids


def capture_vllm(parser_name, cases):
    from vllm.tool_parsers import ToolParserManager
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    parser_cls = ToolParserManager.get_tool_parser(parser_name)

    # Minimal tokenizer stub. Most vLLM tool parsers regex over accumulated text,
    # but DeepSeek V3/V3.1 streaming also counts special marker token IDs. Give
    # those markers stable synthetic IDs so capture can exercise the parser
    # without a model tokenizer.
    class MockTok:
        all_special_tokens = list(_SPECIAL_TOKEN_IDS)
        vocab_size = len(_SPECIAL_TOKEN_IDS)

        def get_vocab(self):
            return _SPECIAL_TOKEN_IDS

        def convert_ids_to_tokens(self, ids, **kw):
            by_id = {v: k for k, v in _SPECIAL_TOKEN_IDS.items()}
            return [by_id.get(i, "") for i in ids]

        def convert_tokens_to_ids(self, tokens):
            if isinstance(tokens, str):
                return _SPECIAL_TOKEN_IDS.get(tokens)
            return [_SPECIAL_TOKEN_IDS.get(token) for token in tokens]

        def decode(self, ids, **kw):
            return ""

        def encode(self, text, **kw):
            return _synthetic_token_ids(text)

    out = {}
    for cid, case in cases.items():
        tools = [
            {"type": "function", "function": {"name": t["name"],
             "parameters": t.get("parameters", {})}}
            for t in (case.get("tools") or [])
        ]
        req = ChatCompletionRequest(model="x", messages=[], tools=tools)
        parser = parser_cls(MockTok())
        prev = ""
        prev_token_ids = []
        per_chunk = []
        for chunk in case.get("chunks", []):
            delta = chunk.get("delta_text", "")
            cur = prev + delta
            delta_token_ids = chunk.get("delta_token_ids") or _synthetic_token_ids(delta)
            current_token_ids = prev_token_ids + delta_token_ids
            try:
                r = parser.extract_tool_calls_streaming(
                    prev, cur, delta, prev_token_ids, current_token_ids,
                    delta_token_ids, req)
            except Exception as e:
                per_chunk.append({"deltas": [], "normal_text": "", "error": str(e)})
                prev = cur
                prev_token_ids = current_token_ids
                continue
            deltas = []
            normal = ""
            if r is not None:
                if getattr(r, "content", None):
                    normal = r.content
                for tc in (r.tool_calls or []):
                    d = {"index": tc.index}
                    if tc.id is not None:
                        d["id"] = True
                    fn = tc.function
                    if fn is not None:
                        if fn.name is not None:
                            d["name"] = fn.name
                        if fn.arguments is not None:
                            d["arguments"] = fn.arguments
                    deltas.append(d)
            per_chunk.append({"deltas": deltas, "normal_text": normal})
            prev = cur
            prev_token_ids = current_token_ids
        out[cid] = per_chunk
    return out


def capture_sglang(parser_name, cases):
    sys.path.insert(0, "/sgl-workspace/sglang/python")
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.entrypoints.openai.protocol import Tool, Function

    detector_cls = FunctionCallParser.ToolCallParserEnum[parser_name]

    out = {}
    for cid, case in cases.items():
        tools = [
            Tool(type="function", function=Function(
                name=t["name"], parameters=t.get("parameters", {})))
            for t in (case.get("tools") or [])
        ]
        det = detector_cls()
        per_chunk = []
        for chunk in case.get("chunks", []):
            delta = chunk.get("delta_text", "")
            try:
                r = det.parse_streaming_increment(delta, tools)
            except Exception as e:
                per_chunk.append({"deltas": [], "normal_text": "", "error": str(e)})
                continue
            deltas = []
            for c in (r.calls or []):
                d = {"index": c.tool_index}
                if c.name:
                    d["name"] = c.name
                if c.parameters:
                    d["arguments"] = c.parameters
                deltas.append(d)
            per_chunk.append({"deltas": deltas, "normal_text": r.normal_text or ""})
        out[cid] = per_chunk
    return out


def engine_version(impl):
    if impl == "vllm":
        import vllm
        return vllm.__version__
    sys.path.insert(0, "/sgl-workspace/sglang/python")
    import sglang
    return sglang.__version__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, choices=("vllm", "sglang"))
    ap.add_argument("--fixture", help="single fixture path")
    ap.add_argument("--parser", help="parser/detector name (single mode)")
    ap.add_argument("--batch", help="JSON: [{fixture, parser}, ...] (batch mode)")
    args = ap.parse_args()

    fn = capture_vllm if args.impl == "vllm" else capture_sglang
    version = engine_version(args.impl)

    if args.batch:
        jobs = json.loads(args.batch)
        out = {}
        for job in jobs:
            doc = yaml.safe_load(open(job["fixture"]))
            try:
                cases = fn(job["parser"], doc.get("cases", {}))
                out[job["fixture"]] = {"cases": cases}
            except Exception as e:
                out[job["fixture"]] = {"error": str(e)}
        print(json.dumps({"version": version, "fixtures": out}, ensure_ascii=False))
        return

    doc = yaml.safe_load(open(args.fixture))
    result = fn(args.parser, doc.get("cases", {}))
    print(json.dumps({"version": version, "cases": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
