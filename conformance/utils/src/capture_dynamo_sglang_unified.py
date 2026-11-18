# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture Dynamo's SGLang chat processor (`--dyn-chat-processor sglang`).

The processor wraps SGLang's reasoning and function-call detectors in its own
post-processing, which decides when tool calls are emitted and which are kept.
Each authored chunk becomes one synthetic token, so the published
`SglangStreamingPostProcessor.process_output` decodes, filters and parses the
exact chunk schedule. Jobs, registration gates and outputs match the SGLang
stream adapter; only the component under test differs.
"""
import json
import sys
import traceback
from importlib.metadata import version

import yaml

from capture_stimulus import capture_peer_results, validate_family_scoped_cases
from impls import DYNAMO_SGLANG_UNIFIED_PIPELINES as FAMILY_PARSERS
from unified_tools import unified_tools

from peer_modules import resolve_peer_module, parser_error_detail, peer_call

processor = resolve_peer_module("dynamo_sglang", "processor")
FunctionCallParser = resolve_peer_module("sglang", "tools").FunctionCallParser
ReasoningParser = resolve_peer_module("sglang", "reasoning").ReasoningParser

TOOL_SCHEMAS = unified_tools()


class ChunkTokenizer:
    """One synthetic token per authored chunk; decoding returns the chunk text.

    No delimiter encodes to a single token, so processors that gate tool
    parsing on delimiter token ids keep their text path.
    """

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def decode(self, token_ids, skip_special_tokens=False, **_kwargs):
        return "".join(self._chunks[token_id] for token_id in token_ids)

    def encode(self, _text, **_kwargs):
        return []


def _registration_unavailable(case):
    if case.get("finish_reason", "stop") not in {"stop", "length"}:
        return "peer_termination_unsupported", "Dynamo SGLang harness supports serving stop/length termination only"
    rname, tname = FAMILY_PARSERS[case["family"]]
    missing = []
    if rname not in ReasoningParser.DetectorMap:
        missing.append(f"ReasoningParser {rname}")
    if tname not in FunctionCallParser.ToolCallParserEnum:
        missing.append(f"FunctionCallParser {tname}")
    if missing:
        return "peer_detector_unavailable", "SGLang " + version("sglang") + " does not register " + ", ".join(missing)
    return None


def _deltas(out):
    """Project one OpenAI choice delta onto stream rows: reasoning, text, then calls."""
    delta = (out or {}).get("delta") or {}
    deltas = [{"kind": "reasoning", "text": delta["reasoning_content"]}] if delta.get("reasoning_content") else []
    if delta.get("content"):
        deltas.append({"kind": "text", "text": delta["content"]})
    for call in delta.get("tool_calls") or []:
        function = call.get("function") or {}
        deltas.append({"kind": "tool_call", "name": function.get("name"), "arguments": function.get("arguments")})
    return deltas


def _stream_chunks(family, chunks, *, terminal_step=False, finish_reason="stop", tools=TOOL_SCHEMAS, rows=None):
    rname, tname = FAMILY_PARSERS[family]
    request = {"tools": [{"type": "function", "function": tool} for tool in tools], "tool_choice": "auto"}
    sglang_tools = peer_call(processor.convert_tools, request["tools"])
    # A native/default request has no template forcing reasoning.
    tool_parser, reasoning_parser = peer_call(
        processor.create_parsers, request, tool_call_parser_name=tname,
        reasoning_parser_name=rname, sglang_tools=sglang_tools, force_reasoning=False)
    post = peer_call(
        processor.SglangStreamingPostProcessor, tokenizer=ChunkTokenizer(chunks),
        tool_call_parser=tool_parser, reasoning_parser=reasoning_parser,
        sglang_tools=sglang_tools, tool_call_parser_name=tname)
    rows = [] if rows is None else rows
    for index in range(len(chunks)):
        final = index == len(chunks) - 1 and not terminal_step
        out = peer_call(post.process_output, {"token_ids": [index], "finish_reason": finish_reason if final else None})
        rows.append(_deltas(out))
    if terminal_step:
        # The authored terminal row is a serving boundary, not model text.
        rows.append(_deltas(peer_call(post.process_output, {"token_ids": [], "finish_reason": finish_reason})))
    return rows


def _capture_cases(cases):
    results = {}
    for case in cases:
        fam = case["family"]
        if fam not in FAMILY_PARSERS:
            continue
        entry = {"parser": "Dynamo SGLang processor " + version("ai-dynamo") + " (stream, SGLang " + version("sglang") + ")"}
        partial_rows = []
        try:
            entry["chunks"] = _stream_chunks(
                fam, case.get("chunks") or [], terminal_step=case.get("terminal_step", False),
                finish_reason=case.get("finish_reason", "stop"), tools=case.get("tools", TOOL_SCHEMAS), rows=partial_rows)
        except Exception as exc:  # noqa: BLE001 — record per-case, keep the batch going
            entry["partial_chunks"] = partial_rows
            entry["error"] = parser_error_detail(exc)
            entry["error_traceback"] = traceback.format_exc()
        results[case["id"]] = entry
    return results


def main():
    job = json.load(sys.stdin)
    cases = job.get("cases", [])
    validate_family_scoped_cases(cases)
    results = {}
    for case in cases:
        results.update(capture_peer_results(
            [case], FAMILY_PARSERS, _capture_cases, tools=case["tools"],
            unsupported_reason=_registration_unavailable,
            supports_finish=True,
        ))
    yaml.dump({"dynamo_sglang_version": version("ai-dynamo"), "sglang_version": version("sglang"),
               "results": results}, sys.stdout,
              default_flow_style=False, sort_keys=False, allow_unicode=True, width=4096)


if __name__ == "__main__":
    main()
