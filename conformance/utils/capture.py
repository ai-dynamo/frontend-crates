#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified in-container capture worker. Runs INSIDE an engine container
(vllm-localdev / sglang-localdev) and drives that engine's tool-call parser.

All engine imports are lazy (inside the per-mode functions) so this one file
imports cleanly in either container — vLLM-only and SGLang-only paths never load
the other engine.

Modes (`--mode`):
  stream         Per-chunk streaming deltas for a stream fixture (TC stream tab).
                   capture.py --mode stream --impl {vllm,sglang} --fixture F --parser P
                   capture.py --mode stream --impl {vllm,sglang} --batch '[{fixture,parser},...]'
                 Emits {version, cases|fixtures: ...}.
  batch-on-stream  Each batch sample's full model_text fed to the streaming parser
                 as one increment, assembled to {calls, normal_text} (batch-on-stream tab).
                   capture.py --mode batch-on-stream --impl {vllm,sglang} --fixture F --parser P
                   capture.py --mode batch-on-stream --impl {vllm,sglang} --batch '[{fixture,parser},...]'
  harmony-batch  Harmony streaming parser over batch model_text, JSON in/out
                 (feeds merge into harmony_batch_stream.json).
                   capture.py --mode harmony-batch --impl {vllm,sglang} --input '{cid:{model_text,tools}}'
                 Emits {cid: {calls}}; engine version to stderr.
  harmony-chunk  vLLM harmony per-chunk, token-native (TC stream tab, harmony). vLLM only.
                   capture.py --mode harmony-chunk --fixture F
                 Emits the bare {cid: [{deltas, normal_text}, ...]} shape; version to stderr.
"""
import argparse
import importlib.metadata as meta
import json
import sys

import yaml


# --------------------------------------------------------------------------- #
# Shared: synthetic special-token ids. Most vLLM tool parsers regex over
# accumulated text, but DeepSeek V3/V3.1 streaming also counts special marker
# token IDs; give those markers stable synthetic IDs so capture can run without a
# model tokenizer.
# --------------------------------------------------------------------------- #
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


def engine_version(impl):
    if impl == "vllm":
        import vllm

        return vllm.__version__
    sys.path.insert(0, "/sgl-workspace/sglang/python")
    import sglang

    return sglang.__version__


# --------------------------------------------------------------------------- #
# mode=stream : per-chunk streaming deltas (was capture_stream.py)
# --------------------------------------------------------------------------- #
def _stream_vllm(parser_name, cases):
    from vllm.tool_parsers import ToolParserManager
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    parser_cls = ToolParserManager.get_tool_parser(parser_name)

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


def _stream_sglang(parser_name, cases):
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
        # SGLang uses tool_index = -1 before it has assigned a stable index. The
        # fixture schema (and cross-engine compare) needs a real u32 index, so
        # track a running index: a -1 delta carrying a name starts a new tool
        # (advance), a -1 delta with only arguments continues the current one.
        cur_index = -1
        for chunk in case.get("chunks", []):
            delta = chunk.get("delta_text", "")
            try:
                r = det.parse_streaming_increment(delta, tools)
            except Exception as e:
                per_chunk.append({"deltas": [], "normal_text": "", "error": str(e)})
                continue
            deltas = []
            for c in (r.calls or []):
                idx = c.tool_index
                if idx is None or idx < 0:
                    if c.name:
                        cur_index += 1
                    idx = cur_index if cur_index >= 0 else 0
                else:
                    cur_index = max(cur_index, idx)
                d = {"index": idx}
                if c.name:
                    d["name"] = c.name
                if c.parameters:
                    d["arguments"] = c.parameters
                deltas.append(d)
            per_chunk.append({"deltas": deltas, "normal_text": r.normal_text or ""})
        out[cid] = per_chunk
    return out


def _run_stream(args):
    fn = _stream_vllm if args.impl == "vllm" else _stream_sglang
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


# --------------------------------------------------------------------------- #
# mode=harmony-batch : harmony streaming parser over batch text, JSON in/out
# (was capture_harmony_batch_stream.py)
# --------------------------------------------------------------------------- #
def _assemble_harmony_deltas(deltas):
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


def _harmony_batch_vllm(cases):
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
        out[cid] = {"calls": _assemble_harmony_deltas(deltas)}
    print(f"openai_harmony {meta.version('openai_harmony')}", file=sys.stderr)
    return out


def _harmony_batch_sglang(cases):
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
        out[cid] = {"calls": _assemble_harmony_deltas(deltas), "normal_text": r.normal_text or ""}
    print(f"sglang {meta.version('sglang')}", file=sys.stderr)
    return out


def _run_harmony_batch(args):
    cases = json.load(open(args.input))
    out = _harmony_batch_vllm(cases) if args.impl == "vllm" else _harmony_batch_sglang(cases)
    print(json.dumps(out, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# mode=batch-on-stream : streaming parser over batch text, all families
# (was capture_batch_on_stream.py — harmony via the harmony-batch path above,
#  others via the per-chunk stream path + assembly)
# --------------------------------------------------------------------------- #
def _assemble_chunks(chunks):
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


def _bos_capture_fixture(impl, parser, fixture):
    doc = yaml.safe_load(open(fixture))
    family = doc["family"]
    cases = doc.get("cases", {})
    if family == "harmony":
        harmony_cases = _harmony_cases(cases)
        fn = _harmony_batch_vllm if impl == "vllm" else _harmony_batch_sglang
        return fn(harmony_cases)

    stream_cases = _batch_cases_to_stream_cases(cases)
    fn = _stream_vllm if impl == "vllm" else _stream_sglang
    per_chunk = fn(parser, stream_cases)
    return {cid: _assemble_chunks(chunks) for cid, chunks in per_chunk.items()}


def _run_batch_on_stream(args):
    if args.batch:
        jobs = json.loads(args.batch)
        fixtures = {}
        for job in jobs:
            try:
                fixtures[job["fixture"]] = {
                    "cases": _bos_capture_fixture(
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
    result = _bos_capture_fixture(args.impl, args.parser, args.fixture)
    print(
        json.dumps(
            {"version": engine_version(args.impl), "cases": result},
            ensure_ascii=False,
        )
    )


# --------------------------------------------------------------------------- #
# mode=harmony-chunk : vLLM harmony per-chunk, token-native (was capture_vllm_harmony.py)
# --------------------------------------------------------------------------- #
def _run_harmony_chunk(args):
    from openai_harmony import HarmonyEncodingName, StreamableParser, load_harmony_encoding
    from vllm.entrypoints.openai.chat_completion.stream_harmony import (
        TokenState,
        extract_harmony_streaming_delta,
    )

    # gpt-oss harmony special tokens: <|start|> = 200006, assistant = 173781.
    # StreamableParser is created in ExpectStart mode (role=None) so it accepts a
    # leading <|start|>. For channel-first inputs (no <|start|>) we prepend
    # <|start|>assistant — the same normalization the Dynamo parser v2 Harmony
    # parser does — so both parsers process the identical token stream.
    START_TOKEN = 200006
    PREAMBLE = [200006, 173781]

    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    doc = yaml.safe_load(open(args.fixture))
    out = {}
    for cid, case in doc.get("cases", {}).items():
        parser = StreamableParser(enc, role=None)
        per_chunk = []
        prepended = False
        broken = False  # parser hit a terminal/unexpected token; stop feeding
        for chunk in case.get("chunks", []):
            ids = list(chunk.get("delta_token_ids", []) or [])
            if not prepended and ids:
                prepended = True
                if ids[0] != START_TOKEN:
                    ids = PREAMBLE + ids
            deltas, normal = [], ""
            for tid in ids:
                if broken:
                    break
                prev_recipient = parser.current_recipient
                try:
                    parser.process(tid)
                except Exception:
                    broken = True
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
                if getattr(dm, "content", None):
                    normal += dm.content
                for tc in (dm.tool_calls or []):
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
        out[cid] = per_chunk
    print(f"openai_harmony {meta.version('openai_harmony')}", file=sys.stderr)
    print(json.dumps(out, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        required=True,
        choices=("stream", "batch-on-stream", "harmony-batch", "harmony-chunk"),
    )
    ap.add_argument("--impl", choices=("vllm", "sglang"))
    ap.add_argument("--fixture", help="single fixture path (YAML)")
    ap.add_argument("--parser", help="parser/detector name (single mode)")
    ap.add_argument("--batch", help="JSON: [{fixture, parser}, ...] (batch mode)")
    ap.add_argument("--input", help="JSON {cid:{model_text,tools}} (harmony-batch)")
    args = ap.parse_args()

    # Per-mode required args (argparse can't express "required only for some modes").
    if args.mode in ("stream", "batch-on-stream", "harmony-batch") and not args.impl:
        ap.error(f"--mode {args.mode} requires --impl {{vllm,sglang}}")
    if args.mode in ("stream", "batch-on-stream") and not (args.fixture or args.batch):
        ap.error(f"--mode {args.mode} requires --fixture or --batch")
    if args.mode == "harmony-batch" and not args.input:
        ap.error("--mode harmony-batch requires --input")
    if args.mode == "harmony-chunk" and not args.fixture:
        ap.error("--mode harmony-chunk requires --fixture")

    if args.mode == "stream":
        _run_stream(args)
    elif args.mode == "batch-on-stream":
        _run_batch_on_stream(args)
    elif args.mode == "harmony-batch":
        _run_harmony_batch(args)
    elif args.mode == "harmony-chunk":
        _run_harmony_chunk(args)


if __name__ == "__main__":
    main()
