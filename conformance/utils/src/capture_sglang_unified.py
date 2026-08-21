# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live-capture SGLang's reasoning + tool detectors for the Unified conformance tab.

SGLang has no unified parser — it always runs a reasoning detector THEN a tool
(function-call) detector, i.e. the Combined/split shape. Runs INSIDE an SGLang
container. Reads a JSON job on stdin ({"cases":[{id,family,input,chunks}]}) and
writes {"sglang_version", "results": {id: {assembled, chunks, parser}}} on stdout.

Batch is the reasoning parser's parse_non_stream(text) -> (reasoning, rest), then the
tool parser's parse_non_stream(rest) -> (content, tool_calls), projected to the ordered
golden event list [reasoning?, content?, tool_calls...] (reasoning always first — the
split can't interleave, which is exactly the divergence the tab measures).
"""
import inspect
import json
import sys
import yaml

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser

# family -> (reasoning model_type, tool_call_parser). Kimi K2.5 uses <think>; SGLang's
# `kimi_k2` reasoning detector matches the golden corpus.
FAMILY_PARSERS = {
    "gemma4": ("gemma4", "gemma4"),
    "qwen3": ("qwen3", "qwen3_coder"),
    "kimi_k2": ("kimi_k2", "kimi_k2"),
    # SGLang names both Muse Glimmer detectors `muse` (PR #34262). vLLM has no
    # released muse parser, so `capture_vllm_unified.py` gets no entry — an entry
    # that cannot capture renders a misleading empty cell instead of an honest gap.
    "muse_glimmer": ("muse", "muse"),
}

TOOLS = [
    Tool(type="function", function=Function(
        name=n, parameters={"type": "object", "properties": {k: {"type": "string"}}}))
    # Must match `tools()` in conformance/tests/unified_parity.rs: an engine that
    # drops calls to unregistered functions would otherwise record a
    # harness-induced divergence for the `log` cases (UNIFIED.12.a / 12.c).
    for n, k in (("get_weather", "city"), ("f", "x"), ("g", "y"), ("run", "cmd"),
                 ("log", "note"))
]


# Here the reasoning parser's normal text always feeds a tool parser, which is exactly
# what `tool_call_parser_active` declares; SGLang's own serving_chat passes it the same
# way. Channel-framed formats (muse) need it: without it the reasoning detector unwraps
# the `to=user` channel, and the tool detector then reads that leading prose as "no
# header is coming", never resyncs on the later tool channel, and drops the call.
#
# The kwarg arrived WITH muse (PR #34262), so released SGLang — including the 0.5.14 that
# captured the committed shard — has no such parameter and raises `TypeError` on it.
# Probe the signature the same way SGLang forwards the flag to its own detectors, or a
# recapture on a muse-less SGLang would error out every case of every family.
_ACTIVE_KWARG = (
    {"tool_call_parser_active": True}
    if "tool_call_parser_active" in inspect.signature(ReasoningParser.__init__).parameters
    else {}
)


def _tool_event(item):
    d = item.model_dump()
    name = d.get("name")
    raw = d.get("parameters")
    if raw is None:
        raw = d.get("arguments")
    try:
        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (ValueError, TypeError):
        args = raw
    return {"kind": "tool_call", "name": name, "arguments": args}


def _tool_stream_delta(item):
    """Streaming tool delta: keep the RAW (possibly partial) argument fragment so the
    consumer can join fragments across chunks — don't json.loads a partial blob."""
    d = item.model_dump()
    return {"kind": "tool_call", "name": d.get("name"),
            "arguments": d.get("parameters") if d.get("parameters") is not None else d.get("arguments")}


def _stream_chunks(family, chunks):
    """SGLang STREAMING: reasoning detector then tool detector, per chunk. Composing the
    two incrementally interleaves reasoning<->tool in order (like Dynamo's streaming)."""
    rname, tname = FAMILY_PARSERS[family]
    rp = ReasoningParser(model_type=rname, **_ACTIVE_KWARG)
    fp = FunctionCallParser(tools=TOOLS, tool_call_parser=tname)
    rows = []
    for ch in chunks:
        deltas = []
        rd, nd = rp.parse_stream_chunk(ch)
        if rd:
            deltas.append({"kind": "reasoning", "text": rd})
        if nd:
            tn, calls = fp.parse_stream_chunk(nd)
            if tn:
                deltas.append({"kind": "text", "text": tn})
            deltas.extend(_tool_stream_delta(c) for c in (calls or []))
        rows.append(deltas)
    return rows


def main():
    job = json.load(sys.stdin)
    import sglang
    results = {}
    for case in job.get("cases", []):
        fam = case["family"]
        if fam not in FAMILY_PARSERS:
            continue
        entry = {"parser": "SGLang Python (Combined)"}
        try:
            entry["chunks"] = _stream_chunks(fam, case.get("chunks") or [])
        except Exception as exc:  # noqa: BLE001 — record per-case, keep the batch going
            entry["chunks"] = []
            entry["error"] = f"{type(exc).__name__}: {exc}"
        results[case["id"]] = entry
    # YAML to match the conformance fixture corpus. Container stdout is log-polluted,
    # so a recapture writes this to a file (or strips lines before the first top-level
    # key) rather than grepping a single JSON line.
    yaml.dump({"sglang_version": getattr(sglang, "__version__", "unknown"),
               "results": results}, sys.stdout,
              default_flow_style=False, sort_keys=False, allow_unicode=True, width=4096)


if __name__ == "__main__":
    main()
